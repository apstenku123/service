# main.tf
provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "rg" {
  name     = "h100-1_group"
  location = "West US 3"
}

resource "azurerm_network_security_group" "nsg" {
  name                = "nc-template-nsg"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_virtual_network" "vnet" {
  name                = "NCASv3-T4-2-vnet"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  address_space       = ["10.0.0.0/16"]
}

resource "azurerm_subnet" "subnet" {
  name                 = "default"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.0.0/24"]
}

resource "azurerm_linux_virtual_machine_scale_set" "spot_instance_scale_set" {
  name                = "nc"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Standard_NC4as_T4_v3"
  instances           = 50
  upgrade_policy_mode = "Manual"
  priority            = "Spot"
  eviction_policy     = "Deallocate"
  max_price           = 0.12

  source_image_reference {
    publisher = "nc.us"
    offer     = "nc_global"
    sku       = "0.0.1"
    version   = "latest"
  }

  os_disk {
    caching           = "ReadWrite"
    managed_disk_type = "Premium_LRS"
  }

  identity {
    type = "SystemAssigned"
  }

  admin_username = "dave"
  admin_password = "Password1234!"

  network_profile {
    name    = "h100-1_group-vnet-nic01"
    primary = true

    ip_configuration {
      name                                    = "internal"
      subnet_id                               = azurerm_subnet.subnet.id
      load_balancer_backend_address_pool_ids  = []
      primary                                 = true
    }
  }

  boot_diagnostics {
    enabled     = true
  }

  custom_data = <<-EOF
              #!/bin/bash
              curl -X GET "http://<MANAGEMENT_SERVER_IP>:<PORT>/get_id" -H "Metadata:true" > /etc/instance_id
              INSTANCE_ID=$(cat /etc/instance_id)
              echo "Instance ID: $INSTANCE_ID" > /var/log/instance_id.log

              # Start application health monitoring
              while true; do
                echo "Health Check: $(date)" > /var/log/health_check.log
                curl -X POST "http://<MANAGEMENT_SERVER_IP>:<PORT>/heartbeat" -d "{\"instance_id\": \"$INSTANCE_ID\"}" -H "Content-Type: application/json"
                sleep 60
              done &
              EOF
}